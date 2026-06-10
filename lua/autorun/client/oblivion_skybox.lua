--[[ oblivion2vmf — custom 3D-skybox backdrop ----------------------------------
  GMod's engine 3D-skybox pass ignores prop_static uniformscale, which leaves the
  baked terrain backdrop floating/misaligned. Instead we draw the model ourselves
  into the sky layer (PostDraw2DSkyBox) using the REAL camera at true world scale,
  so it lines up 1:1 with the playable terrain and shows through the sealed
  toolsskybox walls. Everything is convar-tunable so alignment can be dialled in
  live (no map recompile):

    obliv_sky            1/0   enable
    obliv_sky_scale      16    model scale (the model is baked at 1/16, so 16 = full)
    obliv_sky_z          0     vertical nudge in Hammer units
    obliv_sky_fogstart   8000  backdrop fog start (HU)
    obliv_sky_fogend     60000 backdrop fog end (HU)
-------------------------------------------------------------------------------]]
if SERVER then return end

local MODEL = "models/oblivion2vmf/skybox_terrain.mdl"

CreateClientConVar("obliv_sky",          "1",     true, false, "oblivion2vmf skybox backdrop on/off")
CreateClientConVar("obliv_sky_scale",    "16",    true, false, "backdrop model scale (16 undoes the 1/16 bake)")
CreateClientConVar("obliv_sky_z",        "0",     true, false, "backdrop vertical nudge (HU)")
CreateClientConVar("obliv_sky_fogstart", "8000",  true, false, "backdrop fog start (HU)")
CreateClientConVar("obliv_sky_fogend",   "60000", true, false, "backdrop fog end (HU)")

local mdl

local function build()
    if IsValid(mdl) then mdl:Remove() end
    mdl = nil
    if not file.Exists(MODEL, "GAME") then
        MsgN("[obliv_sky] backdrop model not found: " .. MODEL)
        return
    end
    util.PrecacheModel(MODEL)
    mdl = ClientsideModel(MODEL, RENDERGROUP_OPAQUE)
    if IsValid(mdl) then
        mdl:SetNoDraw(true)                 -- we draw it by hand in the sky pass
    end
end

hook.Add("InitPostEntity", "obliv_sky_build", build)
hook.Add("PostCleanupMap", "obliv_sky_build", build)
build()

-- capture the live render FOV so our cam.Start3D matches the engine's projection
local view_fov = 90
hook.Add("RenderScene", "obliv_sky_fov", function(_, _, fov)
    if fov then view_fov = fov end
end)

hook.Add("PostDraw2DSkyBox", "obliv_sky_draw", function()
    if not GetConVar("obliv_sky"):GetBool() or not IsValid(mdl) then return end

    mdl:SetModelScale(GetConVar("obliv_sky_scale"):GetFloat())
    mdl:SetPos(Vector(0, 0, GetConVar("obliv_sky_z"):GetFloat()))
    mdl:SetAngles(angle_zero)
    mdl:SetupBones()

    -- explicit zNear/zFar: the map's fog-controller farz (16384) otherwise clips
    -- the backdrop, which spans well past that. 200k covers the whole region x16.
    cam.Start3D(EyePos(), EyeAngles(), view_fov, 0, 0, ScrW(), ScrH(), 1, 200000)
        render.SuppressEngineLighting(true)
        render.FogMode(MATERIAL_FOG_LINEAR)
        render.FogStart(GetConVar("obliv_sky_fogstart"):GetFloat())
        render.FogEnd(GetConVar("obliv_sky_fogend"):GetFloat())
        render.FogMaxDensity(1)
        render.FogColor(190, 200, 215)
        render.SetFogZ(-30000)
        mdl:DrawModel()
        render.SuppressEngineLighting(false)
    cam.End3D()
end)
